import asyncio
import socket
import unittest
from unittest import mock

import agent
import queue_runner
from routes import inline_apply
from services import answers


def field(fid, label, kind="input", required=True, options=None, value=""):
    return {"id": fid, "label": label, "kind": kind, "required": required, "options": options or [], "value": value}


class SandboxTest(unittest.TestCase):
    def test_network_is_blocked(self):
        with self.assertRaisesRegex(RuntimeError, "Network access is disabled"):
            socket.create_connection(("example.com", 80))

    def test_no_llm_or_supabase_configured(self):
        self.assertFalse(agent.llm_configured())
        self.assertIsNone(answers.chat.llm_config())
        self.assertIsNone(answers.admin_config())


class PersonalSiteTest(unittest.TestCase):
    def test_keeps_real_sites_and_profile_links(self):
        for url in ("https://sabin.dev", "x.com/sabin", "github.com/sab333n", "linkedin.com/in/someone"):
            self.assertEqual(agent.personal_site(url), url)

    def test_blanks_social_homepages_and_junk(self):
        for url in ("x.com", "https://x.com/", "www.x.com", "github.com", "not a url", "localhost", "", None):
            self.assertEqual(agent.personal_site(url), "", url)


class AskUserTest(unittest.TestCase):
    def test_matches_questions_only_the_applicant_can_answer(self):
        for q in (
            "How do you pronounce your name? (e.g., 'An-na' or 'Ah-na')",
            "Undergraduate GPA",
            "Do you have a Duolingo account? If yes, what is your username?",
            "Have you ever been convicted of a felony?",
            "What is your GitHub handle?",
        ):
            self.assertTrue(agent.ASK_USER.search(q), q)

    def test_leaves_ordinary_questions_alone(self):
        for q in ("Are you legally authorized to work in the United States?", "Why do you want to work here?",
                  "Email", "How did you hear about this job?"):
            self.assertFalse(agent.ASK_USER.search(q), q)


class DecideYesNoTest(unittest.TestCase):
    def setUp(self):
        patcher = mock.patch.dict(agent.PROFILE, {"work_auth": "Yes", "sponsorship": "No", "relocation": "Yes"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_answers_from_profile(self):
        self.assertEqual(agent.decide_yes_no("Are you authorized to work in the US?"), "Yes")
        self.assertEqual(agent.decide_yes_no("Will you now or in the future require sponsorship?"), "No")
        self.assertEqual(agent.decide_yes_no("Have you previously worked at Acme?"), "No")

    def test_never_guesses_personal_history(self):
        self.assertIsNone(agent.decide_yes_no("Have you ever been convicted of a felony?"))

    def test_blank_profile_fields_leave_the_question_unanswered(self):
        with mock.patch.dict(agent.PROFILE, {"work_auth": "", "sponsorship": "", "relocation": ""}):
            self.assertIsNone(agent.decide_yes_no("Are you authorized to work in the US?"))
            self.assertIsNone(agent.decide_yes_no("Will you now or in the future require sponsorship?"))
            self.assertIsNone(agent.decide_yes_no("Are you willing to relocate?"))

    def test_no_values_still_answer_no(self):
        with mock.patch.dict(agent.PROFILE, {"work_auth": "No", "sponsorship": "Yes", "relocation": "no"}):
            self.assertEqual(agent.decide_yes_no("Are you authorized to work in the US?"), "No")
            self.assertEqual(agent.decide_yes_no("Do you require visa sponsorship?"), "Yes")
            self.assertEqual(agent.decide_yes_no("Are you willing to relocate?"), "No")

    def test_ask_grok_skips_the_model_for_ask_user_questions(self):
        with mock.patch.object(agent, "grok_complete", side_effect=AssertionError("model called")):
            self.assertEqual(agent.ask_grok("Undergraduate GPA"), "")


class QueueRunnerTest(unittest.TestCase):
    def test_graduation_question_reads_profile(self):
        # Regression: `grad` was used before assignment and raised NameError.
        with mock.patch.dict(queue_runner.PROFILE, {"graduation_year": "2028"}):
            self.assertEqual(queue_runner.question_answer("What is your expected graduation timeline?"), "2028")

    def test_blank_profile_fields_leave_sentence_questions_unanswered(self):
        blank = {"work_auth": "", "sponsorship": "", "relocation": "", "visa_status": ""}
        with mock.patch.dict(queue_runner.PROFILE, blank), mock.patch.dict(agent.PROFILE, blank):
            for q in ("Are you legally authorized to work in the United States?",
                      "Will you require H-1B sponsorship now or in the future?",
                      "Are you willing to work from the office 3 days a week?"):
                self.assertIsNone(queue_runner.question_answer(q), q)

    def test_ai_answer_never_asks_the_model_about_ask_user_fields(self):
        with mock.patch.object(queue_runner.agent, "grok_complete", side_effect=AssertionError("model called")):
            self.assertEqual(queue_runner.ai_answer("Do you have a Duolingo account?", {"company": "Duolingo"}), "")


class InlineProfileTest(unittest.TestCase):
    def test_use_profile_drops_junk_website(self):
        inline_apply._use_profile({"first_name": "Sabin", "portfolio": "x.com"}, None)
        self.assertEqual(queue_runner.PROFILE["portfolio"], "")
        self.assertEqual(queue_runner.PROFILE["first_name"], "Sabin")


class AnswerHelpersTest(unittest.TestCase):
    def test_placeholder_values(self):
        for v in ("N/A", "n/a", "None", "Not Specified", "-", "TBD"):
            self.assertTrue(answers.PLACEHOLDER.match(v), v)
        self.assertFalse(answers.PLACEHOLDER.match("3.7"))

    def test_is_essay(self):
        self.assertTrue(answers._is_essay(field("a", "Why are you interested in Duolingo?", "textarea")))
        self.assertTrue(answers._is_essay(field("b", "What excites you about this role?")))
        self.assertFalse(answers._is_essay(field("c", "Undergraduate GPA", "textarea")))
        self.assertFalse(answers._is_essay(field("d", "Preferred name, if different?")))
        self.assertFalse(answers._is_essay(field("e", "Why us?", "select", options=["A", "B"])))

    def test_clean_maps_answers_onto_options(self):
        self.assertEqual(answers._clean(field("a", "Q", "select", options=["Yes", "No"]), "yes"), "Yes")
        self.assertIsNone(answers._clean(field("a", "Q", "select", options=["Yes", "No"]), "Maybe"))
        self.assertEqual(answers._clean(field("b", "Agree", "checkbox"), "Yes"), "true")
        self.assertEqual(
            answers._clean(field("c", "Terms", "checkboxes", options=["Summer 2027", "Fall 2027", "Spring 2028"]),
                           "summer 2027, Spring 2028"),
            "Summer 2027, Spring 2028",
        )
        self.assertIsNone(answers._clean(field("d", "Q"), "  "))


class FallbackTest(unittest.TestCase):
    profile = {**agent._PROFILE_DEFAULTS, "first_name": "Sabin", "github": "https://github.com/sab333n",
               "work_auth": "Yes", "gpa": "Not Specified"}
    job = {"company": "Duolingo", "role": "Data Scientist, Intern", "location": "Pittsburgh, PA"}

    def hint(self, f, profile=None):
        return answers._fallback(f, self.job, profile or self.profile)

    def test_fills_facts_from_profile(self):
        self.assertEqual(self.hint(field("a", "Preferred First Name"))["answer"], "Sabin")
        self.assertEqual(self.hint(field("b", "Website"))["answer"], "https://github.com/sab333n")
        auth = self.hint(field("c", "Are you legally authorized to work in the United States?", "select",
                               options=["Yes", "No"]))
        self.assertEqual(auth["answer"], "Yes")

    def test_blank_work_auth_sponsorship_relocation_stay_blank(self):
        blank = {**self.profile, "work_auth": "", "sponsorship": "", "relocation": ""}
        for label in ("Are you legally authorized to work in the United States?",
                      "Will you require sponsorship?", "Are you willing to relocate?"):
            self.assertIsNone(self.hint(field("q", label, "select", options=["Yes", "No"]), blank), label)

    def test_not_authorized_text_answers_no(self):
        hint = self.hint(field("q", "Are you legally authorized to work in the US?", "select", options=["Yes", "No"]),
                         {**self.profile, "work_auth": "Not authorized"})
        self.assertEqual(hint["answer"], "No")

    def test_gpa_only_from_a_real_profile_value(self):
        self.assertIsNone(self.hint(field("a", "Undergraduate GPA")))
        self.assertEqual(self.hint(field("a", "Undergraduate GPA"), {**self.profile, "gpa": "3.7"})["answer"], "3.7")

    def test_never_guesses(self):
        for f in (
            field("a", "How do you pronounce your name?"),
            field("b", "Do you have a Duolingo account? If yes, what is your username?"),
            field("c", "Have you ever been convicted of a felony?", "select", options=["Yes", "No"]),
            field("d", "Do you have a security clearance?", "select", options=["Yes", "No"]),
            field("e", "Why are you interested in Duolingo?", "textarea"),
        ):
            self.assertIsNone(self.hint(f), f["label"])


class SuggestWithoutModelTest(unittest.TestCase):
    def test_fills_facts_and_leaves_the_rest_blank(self):
        profile = {**FallbackTest.profile}
        fields = [
            field("name", "Preferred First Name"),
            field("gpa", "Undergraduate GPA"),
            field("pron", "How do you pronounce your name?"),
            field("why", "Why are you interested in Duolingo?", "textarea"),
            field("impact", "How will your personal experiences make an impact at Duolingo?", "textarea"),
            field("felony", "Have you ever been convicted of a felony?", "select", options=["Yes", "No"]),
            field("done", "Email", value="already@filled.com"),
        ]
        out = asyncio.run(answers.suggest(fields, FallbackTest.job, profile, "local"))
        self.assertEqual(out, {"name": {"answer": "Sabin", "source": "fact"}})


if __name__ == "__main__":
    unittest.main()
